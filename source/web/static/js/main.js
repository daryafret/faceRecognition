$(document).ready(function() {

    $('#file-picker').change(function(){
        var files = $('#form')[0][0].files;
        var existImg = files.length;
        if(existImg == 0) return;
        var form_data = new FormData($('#form')[0]);
        $('#file-picker-label').text(files[0].name);
        $('#loader').show();
        $('#mask').show();
        $.ajax({
                    type: 'POST',
                    url: '/',
                    data: form_data,
                    contentType: false,
                    cache: false,
                    processData: false,
                    success: function(data) {
                        if(data.imgs.length > 0){
                            $('#imgs-results').show();
                            Results.putInto("results-source", "Init img:", data.imgs[0], data, PutDataType.SIMPLE);
                            Results.putInto("results-bb", "With bounding boxes:", data.imgs[1], data, PutDataType.SIMPLE);
                            Results.putInto("inner-result-kl", "Found faces:", data.imgs[2], data, PutDataType.KEYLABEL);
                            Results.putInto("inner-result-rot", "Aligned faces:", data.imgs[3], data, PutDataType.ROTATION);
                            Results.putInto("results-mark", "Final img:", data.imgs[4], data, PutDataType.SIMPLE);

                            var metricPerform = data.metrics.time;
                            metricPerform = Math.round(metricPerform * 100) / 10000;
                            $('#results-metrics').html('<p class="lead text-md-left">Time: ' + metricPerform + ' sec.</p>');
                            $('#results-metrics').show();
                        }
                        $('#loader').hide();
                        $('#mask').hide();
                    },
                    error: function (jqXHR, exception) {
                      $('#loader').hide();
                      $('#mask').hide();
                      Notify.generate('Image recognition has failed.', 'Error');
                    }
         });
    });
    Notify = {
        generate : function (textT, headerT){
            var wtype = 'alert-danger';
            var text = '';
            if(headerT){
                text += "<h4>" + headerT + "</h4>";
            }
            text += "<p>" + textT + "</p>";
            var notif = $("<div class='alert " + wtype + "'><button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button>"+text+"</div>");
            setTimeout(function () {
                    notif.alert('close');
                }, 3000);
            notif.appendTo($("#notifies"));
        }
    }

    PutDataType = {SIMPLE: 1, KEYLABEL: 2, ROTATION: 3}

    Results = {
        putInto  : function(tegId, title, pathToMainImage, pathToRoot, type){ // type = simple, keylabel, rotation
            var elementHtml = '';
            elementHtml += '<p class="lead">' + title + '</p>';
            if(type == PutDataType.SIMPLE){
                elementHtml += Results.getImgItem(pathToMainImage, '', '');
            } else {
                elementHtml += Results.getTopHtmlCarousel(pathToRoot, type);
            }
            $('#'+ tegId).html(elementHtml);
        },

        getTopHtmlCarousel : function(pathToRoot, type){
            var array;
            if(type == PutDataType.KEYLABEL){
                array = pathToRoot.kl;
            } else if (type == PutDataType.ROTATION){
                array = pathToRoot.rot;
            }
            return Results.getHtmlCarousel(array);
        },

        getHtmlCarousel : function (pathsToImgs) {
            var html = '<div class="carousel">';
            var i;
            for(i = 0; i < pathsToImgs.length; i++){
                html += Results.getItemHtml(i, pathsToImgs[i]);
            }
            html += '</div>';
            return html;
        },

        getItemHtml : function (position, pathToImg) {
            
                return Results.getImgItem(pathToImg,["d-block", "mx-auto", "w-70", "h-75"], ["w-50"]);
        },

        getImgItem : function (pathToImg, stylesToImg, stylesToTarget){
            var fullStylesToImg = '';
            var fullStylesToTarget = '';
            var fullItem = '';
            for(var i in stylesToImg) {
                fullStylesToImg += stylesToImg[i] + ' ';
            }

            for(var i in stylesToTarget) {
                fullStylesToTarget += stylesToTarget[i] + ' ';
            }
 	    
            var hash = Results.hashString(pathToImg);
            fullItem += '<a class="miniBox" href="#'+ hash +'"><img src="'+ pathToImg + '" class="'+ fullStylesToImg +'"/></a><div class="miniBox-target" id="'+ hash +'">'
                     + '<a href="#'+ hash +'r"> <img src="'+ pathToImg + '" class="'+ fullStylesToTarget +'"/></a></div>';
            
            return fullItem;

        },
        hashString : function (str){
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                hash += Math.pow(str.charCodeAt(i) * 31, str.length - i);
                hash = hash & hash;
            }
            return hash;
        }
    }
});

